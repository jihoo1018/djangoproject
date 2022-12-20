from django.urls import re_path as url
from admin.dlearn.webcrawler import views

urlpatterns = [
    url(r'naver-movie', views.navermovie)
]