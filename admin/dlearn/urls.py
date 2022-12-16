from django.urls import path, re_path as url
from admin.dlearn import fashion_view

urlpatterns = [
    url(r'fashion2/(?P<num>)$', fashion_view.fashion),
    url(r'fashion', fashion_view.fashion)
]