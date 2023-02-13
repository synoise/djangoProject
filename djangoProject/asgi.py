"""
ASGI config for djangoProject project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

from django.urls import re_path
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

import djangoProject
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AllowedHostsOriginValidator(
            AuthMiddlewareStack(URLRouter(
                [
                    re_path("ws/chat/array", djangoProject.ChatConsumer.as_asgi()),
                    re_path("ws/chat/learn", djangoProject.LearnConsumer.as_asgi()),
                    # re_path(r"ws/chat/(?P<room_name>\w+)/$", djangoProject.ChatConsumer.as_asgi()),
                ]
            ))
        )
    }
)