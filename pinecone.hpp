#pragma once

#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <map>
#include <mutex>
#include <thread>
#include <variant>
#include <vector>

namespace pinecone
{
struct fire_and_forget
{
    struct promise_type
    {
        using handle_type = std::coroutine_handle<promise_type>;

        fire_and_forget    get_return_object() noexcept { return {}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void               return_void() noexcept {}
        void               unhandled_exception() noexcept {}
    };
};

template<typename T>
concept co_mechanism = requires(T a, int t, std::vector<std::coroutine_handle<>> v) {
    {
        a.notify()
    } -> std::same_as<void>;
    {
        a.wait(t, v)
    } -> std::same_as<bool>;
};

template<co_mechanism M>
struct executor
{
    static thread_local executor* thread_executor;

    std::chrono::steady_clock::time_point this_tick = std::chrono::steady_clock::now();

    static void same_thread_resume(std::coroutine_handle<> h)
    {
        // we know we're already on the 'correct' thread, so that given
        // we can either resume immediately, or schedule the resume.
        // theoretically resuming immediately is better, less wasteful
        // boilerplate, however that could lead to long stretches of
        // logic which don't allow the executor to tick, so instead
        // conditionally resume directly, based on how much time has
        // already passed on this tick
        auto  now = std::chrono::steady_clock::now();
        auto* e   = thread_executor;
        if ((now - e->this_tick) > std::chrono::milliseconds{10})
        {
            // resuming on our executor for this thread, so we don't need to notify
            // as we'll check at the beginning of the next tick
            std::unique_lock l{e->m_};
            e->resumable_tasks_.push_back(std::move(h));
        }
        else
        {
            h.resume();
        }
    }

    void cross_thread_resume(std::coroutine_handle<> h)
    {
        {
            std::unique_lock l{m_};
            resumable_tasks_.push_back(std::move(h));
        }

        mechanism_.notify();
    }

    template<typename Rep, typename Period>
    auto same_thread_schedule(std::coroutine_handle<> h, std::chrono::duration<Rep, Period> d)
    {
        auto t = std::chrono::steady_clock::now() + d;
        scheduled_tasks_.emplace(std::move(t), std::move(h));

        // we don't need to notify, we only support scheduling from the same thread
        // so we know we're not currently in a wait
    }

    bool has_tasks() const
    {
        return !scheduled_tasks_.empty() && [&]() {
            std::unique_lock l{m_};
            return !resumable_tasks_.empty();
        }();
    }

    void destroy_tasks()
    {
        for (auto& [time, t] : scheduled_tasks_)
            t.destroy();

        std::unique_lock l{m_};
        for (auto& t : resumable_tasks_)
            t.destroy();
    }

    void wait_for_tasks(std::vector<std::coroutine_handle<>>& to_resume)
    {
        // we're responsible for clearing/ignoring to_resume starting contents
        to_resume.clear(); // keep memory so we don't have to constantly allocate, just clear

        if (scheduled_tasks_.empty())
        {
            // wait for any event, for however long (-1 timeout)
            mechanism_.wait(-1, to_resume);
        }
        else
        {
            // we have ms resolution here, so add one ms (the future) and use lower_bound
            // so we don't get equal elements
            auto now         = std::chrono::steady_clock::now();
            auto lower_bound = scheduled_tasks_.lower_bound(now + std::chrono::milliseconds{1});
            auto start       = scheduled_tasks_.begin();

            {
                // we only need to wait if no scheduled tasks are ready
                if (lower_bound == start)
                {
                    // nothing is ready, but we need to wait no longer than until the first scheduled task
                    int timeout_ms = static_cast<int>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(lower_bound->first - now).count());
                    if (!mechanism_.wait(timeout_ms, to_resume))
                    {
                        // timeout means we should schedule at least the first task
                        // get the new upper bound
                        lower_bound =
                            scheduled_tasks_.lower_bound(std::chrono::steady_clock::now() + std::chrono::milliseconds{1});
                    }
                }

                // potentially schedule tasks ready to go
                std::unique_lock l{m_};
                for (auto it = start; it != lower_bound; ++it)
                {
                    resumable_tasks_.push_back(std::move(it->second));
                }
            }

            scheduled_tasks_.erase(start, lower_bound);
        }

        std::unique_lock l{m_};
        std::move(resumable_tasks_.begin(), resumable_tasks_.end(), std::back_inserter(to_resume));
        resumable_tasks_.clear(); // clear to keep memory allocation
    }

    auto available_executor()
    {
        struct awaiter
        {
            executor* e;
            bool      await_ready() const noexcept { return false; }
            void      await_suspend(std::coroutine_handle<> h) { e->cross_thread_resume(std::move(h)); }
            void      await_resume() const noexcept {}
        };

        return awaiter{this};
    }

    template<typename Rep, typename Period>
    auto wait(std::chrono::duration<Rep, Period> d)
    {
        // only safe to call on our thread, no lock needed
        struct awaiter
        {
            std::chrono::duration<Rep, Period> d;
            executor*                          e;
            bool                               await_ready() const noexcept { return false; }
            void await_suspend(std::coroutine_handle<> h) { e->same_thread_schedule(std::move(h), std::move(d)); }
            void await_resume() const noexcept {}
        };

        return awaiter{std::move(d), this};
    }

    template<typename F, typename... Args>
    void schedule(F&& f, Args&&... args)
    {
        // orphan a coroutine in this function (which _IS NOT A COROUTINE_)
        // we immediately await executor availability which will return
        // code flow here into a fire_and_forget object which we discard.
        // It will then destruct but it doesn't destroy the coroutine handle,
        // which gets resumed by the executor on its thread, and will
        // call the function. We can't await the object or use it again
        // so it destructing is irrelevant. fire_and_forget does not suspend
        // in final_suspend, and as such the coroutine handle cleans up when
        // the function finishes
        [](F&& f, executor<M>* e, Args&&... args) -> fire_and_forget {
            co_await e->available_executor(); // suspend, we need to be resume on the appropriate thread
            co_await std::forward<F>(f)(
                std::forward<Args>(args)...); // exception could be lost here, trapped in ignored future
        }(std::forward<F>(f), this, std::forward<Args>(args)...);
    }

    struct abandon_orphans_tag
    {
    };

    template<typename F, typename... Args>
    auto execute(F&& f, Args&&... args)
    {
        execute_impl(true, std::forward<F>(f), std::forward<Args>(args)...);
    }

    template<typename F, typename... Args>
    auto execute(abandon_orphans_tag, F&& f, Args&&... args)
    {
        execute_impl(false, std::forward<F>(f), std::forward<Args>(args)...);
    }

    template<typename F, typename... Args>
    auto execute_impl(bool await_orphaned_tasks, F&& f, Args&&... args)
    {
        thread_executor = this;

        // f is a coroutine
        auto f_task = std::forward<F>(f)(std::forward<Args>(args)...);

        std::vector<std::coroutine_handle<>> to_resume;

        while (!f_task.done() || (await_orphaned_tasks && has_tasks()))
        {
            // wait_for_tasks will clear to_resume, ignoring anything already in the vector
            wait_for_tasks(to_resume);

            // update tick
            this_tick = std::chrono::steady_clock::now();
            for (auto& handle : to_resume)
            {
                // TODO: only loop until a certain amount of time has passed.
                // this way we can support task priority and age, focusing on
                // the oldest tasks of the highest priority. If we let a low
                // priority group of tasks execute greedily here without ever
                // checking for new resumable_tasks, we will miss the high priorty
                // tasks completing in the meantime
                // but... how to track task age through a coroutine_handle?
                // how to track priority through a coroutine_handle?
                handle.resume();
            }
        }

        // if there are orphaned tasks at this point we aren't awaiting them, but we need to clean them up
        destroy_tasks();

        thread_executor = nullptr;

        return f_task.await_resume();
    }

    // tasks that can be resumed, protected by m_ as external threads can modify this
    std::vector<std::coroutine_handle<>> resumable_tasks_;

    // tasks scheduled for the future, unprotected as only modified within the executor thread
    std::multimap<std::chrono::steady_clock::time_point, std::coroutine_handle<>> scheduled_tasks_;

    // mutex to protect resumable_tasks
    mutable std::mutex m_;

    // actual mechanism responsible for event asynchronicity
    M mechanism_{*this};
};

template<co_mechanism M>
thread_local executor<M>* executor<M>::thread_executor{nullptr};

template<typename T>
struct promise_value
{
    using DataType = T;

    void return_value(T value) { value_ = std::move(value); }

    std::variant<std::monostate, T, std::exception_ptr> value_{};
};

template<>
struct promise_value<void>
{
    struct engaged_tag
    {
    };

    using DataType = engaged_tag;

    void return_void() { value_.template emplace<engaged_tag>(); }

    std::variant<std::monostate, engaged_tag, std::exception_ptr> value_{};
};

template<co_mechanism M, typename R>
class future
{
public:
    struct promise_type : promise_value<R>
    {
        using Handle = std::coroutine_handle<promise_type>;

        future get_return_object() { return future{Handle::from_promise(*this)}; }

        std::suspend_never initial_suspend() noexcept { return {}; }

        // we need to suspend at the end so the promise lives as long as the
        // containing future, which will manually destroy the coroutine
        auto final_suspend() noexcept
        {
            // return an awaiter that resumes `resume_handle_` so the continuation
            // happens only once our coroutine frame is fully unwound
            struct awaiter
            {
                std::optional<std::coroutine_handle<>> handle_;

                bool await_ready() noexcept { return false; }
                void await_suspend(std::coroutine_handle<>) noexcept
                {
                    // the passed coroutine will never resume, it's UB
                    if (handle_)
                        executor<M>::same_thread_resume(handle_.value());
                }
                void await_resume() noexcept {}
            };
            return awaiter{std::exchange(resume_handle_, {})};
        }

        void unhandled_exception() noexcept { this->value_ = std::current_exception(); }

        bool ready() const noexcept { return !std::holds_alternative<std::monostate>(this->value_); }

        std::optional<std::coroutine_handle<>> resume_handle_;
    };

    explicit future(promise_type::Handle coro) : coro_(std::move(coro)) {}

    future(const future&)            = delete;
    future& operator=(const future&) = delete;

    future(future&& move) noexcept : coro_(std::move(move.coro_)) { move.coro_ = {}; }
    future& operator=(future&& move) noexcept
    {
        if (this != &move)
        {
            if (coro_)
                coro_.destroy();
            coro_ = std::exchange(move.coro_, {});
        }
        return *this;
    }

    ~future()
    {
        // coroutine destroys itself when control reaches end of coroutine
        // but that never happens as we always suspend in promise_type::final_suspend
        destroy();
    }

    void destroy()
    {
        if (coro_)
        {
            coro_.destroy();
        }
    }
    void resume() { coro_.resume(); }

    bool await_ready() const noexcept { return coro_.promise().ready(); }
    void await_suspend(std::coroutine_handle<> h) const
    {
        // need h to resume when coro_.promise() is set
        coro_.promise().resume_handle_ = std::move(h);
    }
    typename promise_type::DataType await_resume() const
    {
        if (coro_)
        {
            auto v = std::move(coro_.promise().value_);

            if (std::holds_alternative<std::exception_ptr>(v))
            {
                std::rethrow_exception(std::get<std::exception_ptr>(v));
            }
            else if (std::holds_alternative<typename promise_type::DataType>(v))
            {
                return std::get<typename promise_type::DataType>(std::move(v));
            }
            else
            {
                throw std::logic_error("await_resume on incomplete future");
            }
        }
        else
        {
            throw std::logic_error("await_resume on invalid future");
        }
    }

    bool done() const { return coro_.done(); }

private:
    promise_type::Handle coro_;
};

/// example basic (and likely quite inefficient) mechanism to drive execution
struct cv_mechanism
{
    void notify()
    {
        std::unique_lock l{executor_m_};
        cv_.notify_one();
    }
    bool wait(int timeout_ms, [[maybe_unused]] std::vector<std::coroutine_handle<>>& to_resume)
    {
        std::unique_lock l{executor_m_};
        if (timeout_ms < 0)
        {
            // no predicate here means we can have a race condition, notify is called before the lock is acquired
            // but before we wait, so we never wake up
            cv_.wait(l, [&]() { return !executor_resumable_tasks_.empty(); });
        }
        else
        {
            return cv_.wait_for(
                l, std::chrono::milliseconds{timeout_ms}, [&]() { return !executor_resumable_tasks_.empty(); });
        }

        return true;
    }

    template<typename T>
    cv_mechanism(T&& executor) : executor_resumable_tasks_{executor.resumable_tasks_}, executor_m_{executor.m_}
    {
    }

    // condition variable to wait and signal changes to resumable_tasks
    std::condition_variable cv_;

    std::vector<std::coroutine_handle<>>& executor_resumable_tasks_;
    std::mutex&                           executor_m_;
};

#ifdef EPOLL_CAPABLE
} // end namespace before includes

#include <arpa/inet.h>
#include <fcntl.h>
#include <resolv.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>

namespace pinecone
{

struct epoll_mechanism
{
    struct registered_event
    {
        int                     fd_;
        std::coroutine_handle<> resume_point_;
        bool                    hangup_{false};
    };

    void notify()
    {
        if (eventfd_write(event_fd_, 1))
            throw std::runtime_error("Could not eventfd_write for epoll_mechanism");
    }

    bool wait(int timeout_ms, [[maybe_unused]] std::vector<std::coroutine_handle<>>& to_resume)
    {
        constexpr std::size_t MAX_EVENTS{1}; // only supporting event_fd_ for now
        epoll_event           events[MAX_EVENTS];

        int ready_count = epoll_wait(epoll_fd_, events, MAX_EVENTS, timeout_ms);

        if (ready_count == 0) // zero ready, no events means timeout
            return false;
        else if (ready_count > 0)
        {
            for (int i = 0; i < ready_count; ++i)
            {
                auto& event = events[i];
                int   fd    = event.data.fd;
                if (event.data.fd == event_fd_ && (event.events & EPOLLIN) == EPOLLIN)
                {
                    // need to read the data
                    eventfd_t dummy; // we don't currently care about the value, just that it was written
                    if (eventfd_read(event_fd_, &dummy)) // todo: this could be spurious wakeup
                        throw std::runtime_error("Could not eventfd_read for epoll_mechanism");
                }
                else
                {
                    auto* ev = static_cast<registered_event*>(event.data.ptr);
                    if (event.events & (EPOLLERR | EPOLLHUP))
                    {
                        // flag error so caller can close
                        ev->hangup_ = true;
                    }

                    // regardless: it's a registered event and the data pointer will be the coroutine_handle
                    to_resume.push_back(std::move(ev->resume_point_));
                }
            }
            // anything ready means we should wakeup now
            return true;
        }
        else
            throw std::runtime_error("Could not epoll_wait with epoll_mechanism");
    }

    int start_listen_ipv4(uint16_t port)
    {
        int fd        = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        int reuseaddr = 1;
        if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr, sizeof(reuseaddr)) < 0)
        {
            close(fd);
            throw std::runtime_error("Could not set socket address reuse in epoll_mechanism");
        }

        if (fd < 0)
        {
            throw std::runtime_error("Could not create socket in epoll_mechanism");
        }

        sockaddr_in addr = {};
        addr.sin_family  = AF_INET;
        addr.sin_port    = htons(port);
        if (inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr.s_addr) == 0)
        {
            close(fd);
            throw std::runtime_error("Could not evaluate 127.0.0.1 in epoll_mechanism");
        }

        if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == -1)
        {
            close(fd);
            throw std::runtime_error(fmt::format("Could not bind in epoll_mechanism: errno: {}", errno));
        }

        const int listen_backlog{128};
        if (listen(fd, listen_backlog) == -1)
        {
            close(fd);
            throw std::runtime_error(fmt::format("Could not listen in epoll_mechanism: errno: {}", errno));
        }

        return fd;
    }

    int start_connect_ipv4(const std::string& host, uint16_t port)
    {
        int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);

        if (fd < 0)
        {
            throw std::runtime_error("Could not create socket in epoll_mechanism");
        }

        sockaddr_in dest = {};
        dest.sin_family  = AF_INET;
        dest.sin_port    = htons(port);
        if (inet_pton(AF_INET, host.data(), &dest.sin_addr.s_addr) == 0)
            throw std::runtime_error("Could not evaluate host name in epoll_mechanism");

        auto connect_result = connect(fd, reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
        if (connect_result == 0 || errno == EINPROGRESS)
        {
            // immediate connect or in progress, either way caller can await readability
            return fd;
        }

        throw std::runtime_error(fmt::format("Could not connect in epoll_mechanism. errno: {}", errno));
    }

    auto readable(int fd)
    {
        struct awaiter
        {
            awaiter(int fd, int epoll_fd) : ev_{fd}, epoll_fd_(epoll_fd) {}
            awaiter(const awaiter&)            = delete;
            awaiter(awaiter&&)                 = delete;
            awaiter& operator=(const awaiter&) = delete;
            awaiter& operator=(awaiter&&)      = delete;

            registered_event ev_;
            int              epoll_fd_;
            bool             await_ready() const noexcept { return false; }
            auto             await_suspend(std::coroutine_handle<> h)
            {
                // awake h when we have some data
                ev_.resume_point_ = std::move(h);
                epoll_event read_event;
                read_event.events   = EPOLLIN | EPOLLONESHOT;
                read_event.data.ptr = static_cast<void*>(&ev_);
                if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, ev_.fd_, &read_event))
                    throw std::runtime_error(
                        fmt::format("Could not add read fd via epoll_ctl in epoll_mechanism: {}", errno));
            }
            bool await_resume() const
            {
                if (ev_.hangup_)
                    return false;

                if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, ev_.fd_, nullptr))
                    if (errno == EBADF) // our fd closed
                        return false;
                    else
                        throw std::runtime_error(fmt::format("Could not epoll_ctl delete in epoll_mechanism: {}", errno));
                return true;
            }
        };

        return awaiter{fd, epoll_fd_};
    }

    auto writeable(int fd)
    {
        struct awaiter
        {
            awaiter(int fd, int epoll_fd) : ev_{fd}, epoll_fd_(epoll_fd) {}
            awaiter(const awaiter&)            = delete;
            awaiter(awaiter&&)                 = delete;
            awaiter& operator=(const awaiter&) = delete;
            awaiter& operator=(awaiter&&)      = delete;

            registered_event ev_;
            int              epoll_fd_;
            bool             await_ready() const noexcept { return false; }
            auto             await_suspend(std::coroutine_handle<> h)
            {
                // awake h when we have some data
                ev_.resume_point_ = std::move(h);
                epoll_event write_event;
                write_event.events   = EPOLLOUT | EPOLLONESHOT;
                write_event.data.ptr = static_cast<void*>(&ev_);
                if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, ev_.fd_, &write_event))
                    throw std::runtime_error(
                        fmt::format("Could not add write fd via epoll_ctl in epoll_mechanism: {}", errno));
            }
            bool await_resume() const
            {
                if (ev_.hangup_)
                    return false;

                if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, ev_.fd_, nullptr))
                    if (errno == EBADF)
                        return false; // our fd closed, not exceptional just return failure
                    else
                        throw std::runtime_error(fmt::format("Could not epoll_ctl delete in epoll_mechanism: {}", errno));
                return true;
            }
        };

        return awaiter{fd, epoll_fd_};
    }

    template<typename T>
    epoll_mechanism([[maybe_unused]] T&& executor)
        : epoll_fd_([]() {
              auto fd = epoll_create1(0);
              if (fd == -1)
                  throw std::runtime_error("Could not epoll_create1 for epoll_mechanism");
              return fd;
          }()),
          event_fd_([]() {
              auto fd = eventfd(0, EFD_NONBLOCK);
              if (fd == -1)
                  throw std::runtime_error("Could not epoll_create1 for epoll_mechanism");
              return fd;
          }())
    {
        epoll_event event_fd_event;
        event_fd_event.events  = EPOLLIN;
        event_fd_event.data.fd = event_fd_;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, event_fd_, &event_fd_event))
            throw std::runtime_error("Could not add event_fd_ to epoll_fd_ for epoll_mechanism");
    }

    ~epoll_mechanism()
    {
        // destructor - can't throw here, so ignore possible errors
        if (event_fd_ != -1)
            close(event_fd_);
        if (epoll_fd_ != -1)
            close(epoll_fd_);
    }

    int epoll_fd_{-1};
    int event_fd_{-1}; // used for waking up for non-fd based reasons
};
#endif

const std::string& get_thread_id()
{
    thread_local std::string tid = []() {
        std::stringstream s;
        s << std::this_thread::get_id();

        return s.str();
    }();

    return tid;
}

template<co_mechanism M, typename F>
    requires(!std::same_as<std::invoke_result_t<F &&>, void>)
future<M, std::invoke_result_t<F&&>> offload_to(executor<M>* e, F&& f)
{
    executor<M>* original_executor = executor<M>::thread_executor;

    // create an awaiter for the value, while spawning on e a routine that populates that
    // awaiter and resume it on original_executor
    struct awaiter
    {
        F&&                                                                         f;
        executor<M>*                                                                original_executor;
        executor<M>*                                                                target_executor;
        std::variant<std::monostate, std::invoke_result_t<F&&>, std::exception_ptr> result;
        bool await_ready() const noexcept { return false; }
        void await_suspend(std::coroutine_handle<> h)
        {
            // orphan this future, target_executor must finish this before original_executor
            // will unblock co_awaiting this awaiter. That co_await will keep this awaiter in
            // scope on the coroutine frame until after it is resumed and result consumed
            [](auto* self, std::coroutine_handle<> h) -> fire_and_forget {
                co_await self->target_executor->available_executor();
                try
                {
                    self->result.template emplace<std::invoke_result_t<F&&>>(std::forward<F>(self->f)());
                }
                catch (...)
                {
                    self->result.template emplace<std::exception_ptr>(std::current_exception());
                }
                self->original_executor->cross_thread_resume(std::move(h));
            }(this, std::move(h));
        }
        void await_resume() const noexcept {}
    };

    awaiter a{std::forward<F>(f), executor<M>::thread_executor, e};
    co_await a;

    // value must be set or exception after a is awaited
    if (std::holds_alternative<std::exception_ptr>(a.result))
        std::rethrow_exception(std::get<std::exception_ptr>(a.result));
    else
        co_return std::get<std::invoke_result_t<F&&>>(std::move(a.result));
}

template<co_mechanism M, typename F>
    requires(std::same_as<std::invoke_result_t<F &&>, void>)
future<M, void> offload_to(executor<M>* e, F&& f)
{
    executor<M>* original_executor = executor<M>::thread_executor;

    // create an awaiter for the value, while spawning on e a routine that populates that
    // awaiter and resume it on original_executor
    struct awaiter
    {
        F&&                               f;
        executor<M>*                      original_executor;
        executor<M>*                      target_executor;
        std::optional<std::exception_ptr> exception;
        bool                              await_ready() const noexcept { return false; }
        void                              await_suspend(std::coroutine_handle<> h)
        {
            // orphan this future, target_executor must finish this before original_executor
            // will unblock co_awaiting this awaiter. That co_await will keep this awaiter in
            // scope on the coroutine frame until after it is resumed and result consumed
            [](auto* self, std::coroutine_handle<> h) -> fire_and_forget {
                co_await self->target_executor->available_executor();
                try
                {
                    std::forward<F>(self->f)();
                }
                catch (...)
                {
                    self->exception.emplace(std::current_exception());
                }
                self->original_executor->cross_thread_resume(std::move(h));
            }(this, std::move(h));
        }
        void await_resume() const noexcept {}
    };

    awaiter a{std::forward<F>(f), executor<M>::thread_executor, e};
    co_await a;

    // exception may be set after a is awaited
    if (a.exception.has_value())
        std::rethrow_exception(a.exception.value());
}

template<co_mechanism M, typename F, typename... Args>
    requires(!std::same_as<std::invoke_result_t<F &&>, void>)
future<M, std::invoke_result_t<F&&>> offload_from(executor<M>* e, F&& f)
{
    // should be called from within a coroutine, needs to create a thread,
    // await that thread's completion, and return the value of the functor
    // executed on that thread
    struct awaiter
    {
        F&&                                                                         f;
        executor<M>*                                                                e;
        std::variant<std::monostate, std::invoke_result_t<F&&>, std::exception_ptr> result;
        std::jthread                                                                offload_thread;
        bool await_ready() const noexcept { return false; }
        void await_suspend(std::coroutine_handle<> h)
        {
            // thread captures this but is last member, so destructs (and joins) first
            offload_thread = std::jthread{[this, h = std::move(h)]() {
                try
                {
                    result.template emplace<std::invoke_result_t<F&&>>(std::forward<F>(f)());
                }
                catch (...)
                {
                    result.template emplace<std::exception_ptr>(std::current_exception());
                }
                e->cross_thread_resume(std::move(h));
            }};
        }
        void await_resume() const noexcept {}
    };

    awaiter a{std::forward<F>(f), e};
    co_await a;

    // value must be set or exception after a is awaited
    if (std::holds_alternative<std::exception_ptr>(a.result))
        std::rethrow_exception(std::get<std::exception_ptr>(a.result));
    else
        co_return std::get<std::invoke_result_t<F&&>>(std::move(a.result));
}

template<co_mechanism M, typename F>
    requires(std::same_as<std::invoke_result_t<F &&>, void>)
future<M, void> offload_from(executor<M>* e, F&& f)
{
    // should be called from within a coroutine, needs to create a thread,
    // await that thread's completion, and return the value of the functor
    // executed on that thread
    struct awaiter
    {
        F&&                               f;
        executor<M>*                      e;
        std::optional<std::exception_ptr> exception;
        std::jthread                      offload_thread;
        bool                              await_ready() const noexcept { return false; }
        void                              await_suspend(std::coroutine_handle<> h)
        {
            // thread captures this but is last member, so destructs (and joins) first
            offload_thread = std::jthread{[this, h = std::move(h)]() {
                try
                {
                    std::forward<F>(f)();
                }
                catch (...)
                {
                    exception.emplace(std::current_exception());
                }
                e->cross_thread_resume(std::move(h));
            }};
        }
        void await_resume() const noexcept {}
    };

    awaiter a{std::forward<F>(f), e};

    // exception may be set after a is awaited
    if (a.exception.has_value())
        std::rethrow_exception(a.exception.value());
}

template<typename F>
struct finally
{
    F functor_;

    ~finally() { functor_(); }
};

template<typename T>
class condition_variable
{
public:
    template<typename... Args>
    condition_variable(Args&&... args) : value_(std::forward<Args>(args)...)
    {
    }

    template<typename F>
    auto update(F&& f)
    {
        std::forward<F>(f)(value_);

        auto end = subscribers_.end();

        // move still-failing subscribers to the beginning
        auto satisfied_subscribers = std::partition(subscribers_.begin(), end, [](auto* s) { return !s->check(); });

        for (auto it = satisfied_subscribers; it != end; ++it)
        {
            (*it)->resume();
        }

        subscribers_.erase(satisfied_subscribers, end);
    }

    template<typename F>
    auto changed(F&& f)
    {
        struct awaiter final : subscriber
        {
            awaiter(F&& f, const T& value, std::vector<subscriber*>& subscribers)
                : f_(std::forward<F>(f)), value_(value), subscribers_(subscribers)
            {
            }
            awaiter(const awaiter&)            = delete;
            awaiter(awaiter&&)                 = delete;
            awaiter& operator=(const awaiter&) = delete;
            awaiter& operator=(awaiter&&)      = delete;

            F                         f_;
            const T&                  value_;
            std::vector<subscriber*>& subscribers_;
            std::coroutine_handle<>   h_;
            bool                      await_ready() const noexcept { return check(); }
            auto                      await_suspend(std::coroutine_handle<> h)
            {
                h_ = std::move(h);
                subscribers_.push_back(this);
            }
            void await_resume() const {}

            bool check() const override { return f_(value_); }
            void resume() override { h_.resume(); }
        };

        return awaiter{std::forward<F>(f), value_, subscribers_};
    }

private:
    struct subscriber
    {
        virtual bool check() const = 0;
        virtual void resume()      = 0;
    };

    T                        value_;
    std::vector<subscriber*> subscribers_;
};

template<co_mechanism M, typename F>
future<M, std::string> read_until(M& mechanism, int fd, F&& f)
{
    std::string           message;
    constexpr std::size_t buffer_size{1024};
    char                  buf[buffer_size];

    for (;;)
    {
        if (!co_await mechanism.readable(fd))
        {
            throw std::runtime_error("Remote disconnected on read");
        }
        auto result = ::read(fd, buf, buffer_size);

        if (result == 0) // EOF
        {
            throw std::runtime_error("Remote EOF on read");
        }
        else if (result == -1 && errno != EAGAIN)
            throw std::runtime_error("Failed to read!");
        else
        {
            message.append(buf, static_cast<std::size_t>(result));

            // if we have a newline now we can break out of the loop
            if (f(message))
                break;
        }
    }

    co_return message;
}

template<co_mechanism M>
future<M, void> write(M& mechanism, int fd, const std::string& message)
{
    ssize_t     num_written{0};
    std::size_t length = message.size();

    while (std::cmp_less(num_written, length))
    {
        if (!co_await mechanism.writeable(fd))
        {
            throw std::runtime_error("Remote disconnected on write");
            co_return;
        }

        auto result = ::write(fd, message.data() + num_written, length - num_written);
        if (result == -1 && errno != EAGAIN)
            throw std::runtime_error("Failed to write!");
        else if (result > 0)
            num_written += result;
    }
}

template<co_mechanism M, typename ContextType>
class task
{
public:
    task(ContextType& context) : context_(context) {}

    virtual future<M, bool> execute() = 0;

protected:
    ContextType& context_;
};

template<typename T, typename M, typename C>
concept impltask_of = co_mechanism<M> && std::derived_from<T, task<M, C>>;

template<co_mechanism M, typename ContextType, typename NewContextType, impltask_of<M, NewContextType> WrappedTask>
class scoped_task : public task<M, ContextType>
{
public:
    scoped_task(ContextType& context) : task<M, ContextType>(context) {}

    future<M, bool> execute() override
    {
        NewContextType c{this->context_};
        WrappedTask    t{c};

        co_return co_await t.execute();
    }
};

template<co_mechanism M, typename ContextType, impltask_of<M, ContextType>... Tasks>
class pipeline : public task<M, ContextType>
{
public:
    pipeline(ContextType& context) : task<M, ContextType>(context), tasks_{Tasks{context}...} {}

    future<M, bool> execute() override { co_return co_await execute_impl(std::index_sequence_for<Tasks...>{}); }

private:
    template<size_t... Is>
    future<M, bool> execute_impl(std::index_sequence<Is...>)
    {
        ((co_await std::get<Is>(tasks_).execute()), ...);
        co_return true; // this pipeline ignored child errors
    }

    std::tuple<Tasks...> tasks_;
};

template<co_mechanism M, typename ContextType, impltask_of<M, ContextType>... Tasks>
class or_pipeline : public task<M, ContextType>
{
public:
    or_pipeline(ContextType& context) : task<M, ContextType>(context), tasks_{Tasks{context}...} {}

    future<M, bool> execute() override { co_return co_await execute_impl(std::index_sequence_for<Tasks...>{}); }

private:
    template<size_t I, size_t... Is>
    future<M, bool> execute_impl(std::index_sequence<I, Is...>)
    {
        auto result = co_await std::get<I>(tasks_).execute();

        if constexpr (sizeof...(Is) > 0)
        {
            if (!result)
            {
                co_return co_await execute_impl(std::index_sequence<Is...>{});
            }
        }

        co_return result;
    }

    future<M, bool> execute_impl(std::index_sequence<>) { co_return false; }

    std::tuple<Tasks...> tasks_;
};

template<co_mechanism M, typename ContextType, impltask_of<M, ContextType>... Tasks>
class and_pipeline : public task<M, ContextType>
{
public:
    and_pipeline(ContextType& context) : task<M, ContextType>(context), tasks_{Tasks{context}...} {}

    future<M, bool> execute() override { co_return co_await execute_impl(std::index_sequence_for<Tasks...>{}); }

private:
    template<size_t I, size_t... Is>
    future<M, bool> execute_impl(std::index_sequence<I, Is...>)
    {
        auto result = co_await std::get<I>(tasks_).execute();

        if constexpr (sizeof...(Is) > 0)
        {
            if (result)
            {
                co_return co_await execute_impl(std::index_sequence<Is...>{});
            }
        }

        co_return result;
    }

    future<M, bool> execute_impl(std::index_sequence<>) { co_return true; }

    std::tuple<Tasks...> tasks_;
};

} // namespace pinecone
