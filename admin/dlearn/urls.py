from django.urls import re_path as url
from admin.dlearn import fashion_view
from admin.dlearn.mnist_number import number_view
from admin.dlearn.webcrawler import views

urlpatterns = [
    url(r'fashion', fashion_view.fashion),
    url(r'MnNumber', number_view.number)
]