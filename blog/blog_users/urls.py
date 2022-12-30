from blog.blog_users import views
from django.urls import re_path as url

urlpatterns = [
    url(r'loginform',views.loginform),
    url(r'user-list',views.user_list),
    url(r'users-list/name',views.user_list_name),
    url(r'login-view',views.blog_user_view)
]

