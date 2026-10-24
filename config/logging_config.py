#structured json logging config
#call setup_logging() once at app startup

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Any

import structlog

_LOG_DIR='logs'
_LOG_FILE=os.path.join(_LOG_DIR,'rag_app.log')

def setup_logging(log_level:str="INFO")->None:
    #configure structlog + stdlib logging for structure json output.
    
    os.makedirs(_LOG_DIR,exist_ok=True)
    
    shared_processors:list[Any]=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    formatter=structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    
    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler=TimedRotatingFileHandler(
        filename=_LOG_FILE,
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    root_logger=logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(getattr(logging,log_level.upper(),logging.INFO))
    
    #silence noisy third-party loggers
    for noisy in ('httpx','httpcore','pinecone','urllib3','sentence_transformers','huggingface_hub'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
def get_logger(name:str)->structlog.stdlib.BoundLogger:
    return structlog.getLogger(name)