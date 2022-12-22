from django.urls import re_path as url

from admin.nlp.samsung_report import views

urlpatterns = [
    url(r'ko-nlp', views.konlp)
]