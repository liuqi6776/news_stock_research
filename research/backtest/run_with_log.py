import logging
import sys
import os

# 设置日志
log_file = r"c:\Users\liuqi\quant_system_v2\research\backtest\backtest_run.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

logger.info("Starting backtest...")

# 导入并运行回测
try:
    import direct_backtest
    logger.info("Module imported successfully")
    direct_backtest.main()
    logger.info("Backtest completed")
except Exception as e:
    logger.error(f"Error: {str(e)}", exc_info=True)
    raise
