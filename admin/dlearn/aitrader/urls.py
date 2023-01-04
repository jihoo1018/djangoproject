from django.urls import re_path as url

from admin.dlearn.aitrader.views import kospi

urlpatterns = [
    url(r'kospi',kospi)
]