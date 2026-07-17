"""
重试机制工具模块

提供统一的 API 调用重试装饰器和函数
"""

import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
import warnings


def retry(
    max_attempts: int = 10,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟增长倍数（指数退避）
        max_delay: 最大延迟时间
        exceptions: 需要捕获重试的异常类型
        on_retry: 重试时的回调函数，接收 (exception, attempt) 参数
    
    Example:
        @retry(max_attempts=5, delay=2)
        def fetch_data():
            return api.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        break
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    # 计算下次延迟时间（指数退避）
                    sleep_time = min(current_delay, max_delay)
                    print(f"    [Retry] Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    current_delay *= backoff
            
            # 超过最大重试次数，抛出最后一个异常
            raise last_exception
        
        return wrapper
    return decorator


def retry_call(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    max_attempts: int = 10,
    delay: float = 7.0,
    on_error: Optional[Callable[[Exception, int], None]] = None
) -> Any:
    """
    函数式重试调用
    
    Args:
        func: 要执行的函数
        args: 位置参数
        kwargs: 关键字参数
        max_attempts: 最大重试次数
        delay: 每次重试的延迟时间
        on_error: 错误回调函数
    
    Returns:
        函数执行结果
    
    Raises:
        Exception: 超过最大重试次数后抛出最后一个异常
    
    Example:
        result = retry_call(api.fetch, args=(date,), max_attempts=5)
    """
    if kwargs is None:
        kwargs = {}
    
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if on_error:
                on_error(e, attempt + 1)
            if attempt < max_attempts - 1:
                print(f"    [Retry] Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    
    raise last_exception


class RetryContext:
    """
    重试上下文管理器
    
    Example:
        with RetryContext(max_attempts=5) as ctx:
            while ctx.should_continue():
                try:
                    result = api.call()
                    break
                except Exception as e:
                    ctx.record_failure(e)
    """
    
    def __init__(self, max_attempts: int = 10, delay: float = 7.0):
        self.max_attempts = max_attempts
        self.delay = delay
        self.attempt = 0
        self.last_exception: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def should_continue(self) -> bool:
        """检查是否应该继续重试"""
        return self.attempt < self.max_attempts
    
    def record_failure(self, exception: Exception):
        """记录失败并等待"""
        self.last_exception = exception
        self.attempt += 1
        if self.attempt < self.max_attempts:
            print(f"    [Retry] Attempt {self.attempt}/{self.max_attempts} failed: {exception}. Retrying in {self.delay}s...")
            time.sleep(self.delay)
    
    def raise_if_failed(self):
        """如果所有重试都失败，抛出最后一个异常"""
        if self.last_exception and self.attempt >= self.max_attempts:
            raise self.last_exception
