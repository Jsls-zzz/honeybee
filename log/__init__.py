from log.loggers import get_logger
from common.log_config import LOG_NAME_SERVICE
from common.monitor_config import mail_recipients


logger = get_logger(name=LOG_NAME_SERVICE,
                    fromaddr='jsls96zhangwei@163.com',
                    toaddr=mail_recipients['test'])
