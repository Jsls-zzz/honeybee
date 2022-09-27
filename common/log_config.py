# log存储文件名
LOG_FILE = 'server.log'

#日志句柄名
LOG_NAME_SERVICE = 'log_service'

# 授权密码
CREDENTIALS_PROD = None

CREDENTIALS_TEST = ('jsls96zhangwei@163.com', 'XVZVCQKGUGNLMTZW')

CREDENTIALS = {'prod': CREDENTIALS_PROD, 'test': CREDENTIALS_TEST}

MAIL_CONFIG = {
    'mail_host': ('smtp.163.com', 25),
    'timeout': 2,
}
