from django.urls import re_path as url
from blog.blog_users import views

urlpatterns = [
    url(r'loginform',views.loginform),
    url(r'user-list',views.user_list)
]

