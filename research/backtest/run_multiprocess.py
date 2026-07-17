import multiprocessing
import sys
import os

def run_backtest():
    # 导入并运行回测
    import direct_backtest
    direct_backtest.main()

if __name__ == '__main__':
    # 使用multiprocessing运行回测
    p = multiprocessing.Process(target=run_backtest)
    p.start()
    p.join()
    print(f"Process finished with exit code: {p.exitcode}")
