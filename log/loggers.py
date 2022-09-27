import os
import socket
import logging
from datetime import datetime, timedelta
from logging.handlers import SMTPHandler
from common.log_config import LOG_FILE, MAIL_CONFIG, CREDENTIALS


def get_local_name():
    """
    获取本机文件
    :return: 主机名称
    """
    name = socket.gethostname()
    return name

def generate_log_file_path(file_name):
    """
    获取日志存储路径
    :param file_name:
    :return: 日志存储路径
    """
    cur_path = os.path.abspath(__name__)
    dir_path = os.path.dirname(cur_path)
    log_file_path = os.path.join(dir_path, 'output', file_name)
    return log_file_path

class GXFilter(logging.Filter):
    """
    自定义日志过滤器
    """
    def filter(self, record):
        record.gx = 'gotion'
        return True

class BatchSMTPHandler(SMTPHandler):
    """
    邮件发送间隔不低于60s
    """
    def __init__(self, min_gaps_s=60, **kwargs):
        SMTPHandler.__init__(self, **kwargs)
        self.min_gaps_s = min_gaps_s
        self.last_sent_at = datetime.now() - timedelta(0, min_gaps_s)

    def emit(self, record):
        now = datetime.now()
        if now >= self.last_sent_at + timedelta(0, self.min_gaps_s):
            try:
                self.last_sent_at = now
                SMTPHandler.emit(self, record)
            except:
                pass

class BufferingHandler(BatchSMTPHandler):
    """
    邮件超过异常数量时预警
    1：日志累计10份，发送邮件
    2：日志累计超过5份且距上一封邮件超过6小时，发送邮件
    """
    def __init__(self, capacity=10, max_gap_h=6, min_gap_s=60, **kwargs):
        BatchSMTPHandler.__init__(self, min_gap_s, **kwargs)
        self.capacity = capacity
        self.max_gap_h = max_gap_h
        self.latest_sent_at = datetime.now()
        self.buffer = []

    def shouldFlush(self):
        now = datetime.now()
        is_timeout = now > self.latest_sent_at + timedelta(0, hours=self.max_gap_h)
        if (is_timeout and (len(self.buffer) >= 5)) or (len(self.buffer) >= self.capacity):
            self.latest_sent_at = now
            return True
        return False

    def emit(self, record):
        self.buffer.append(record)
        if self.shouldFlush():
            error_info = ''
            for item in self.buffer:
                error_info += "\n" + item.msg
            record.msg = error_info
            BatchSMTPHandler.emit(self, record)
            self.flush()

    def flush(self):
        self.acquire()
        try:
            self.buffer.clear()
        finally:
            self.release()

def get_logger(name, fromaddr=None, toaddr=None):
    """
    获取日志句柄
    :param name: 日志句柄名
    :param fromaddr: 日志邮件发送者
    :param toaddr: 日志邮件接收者
    :return: 生成的日志实例
    """
    if not name:
        name = 'HandlerLogger'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(GXFilter())

    log_formatter = logging.Formatter(
        fmt='%(gx)s %(asctime)s [%(levelname)s] %(filename)s[line:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    log_file_path = generate_log_file_path(LOG_FILE)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.WARN)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.WARN)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    mail_handler = BufferingHandler(
        mailhost=MAIL_CONFIG['mail_host'],
        # timeout=MAIL_CONFIG['timeout'],
        fromaddr=fromaddr,
        toaddrs=toaddr,
        subject='[CRITICAL] {} from {}'.format(name, get_local_name()),
        credentials=CREDENTIALS['test'],
    )
    mail_handler.setLevel(logging.CRITICAL)
    mail_handler.setFormatter(log_formatter)
    logger.addHandler(mail_handler)

    return logger
