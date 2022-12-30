from django.urls import re_path as url

from blog.view import views

urlpatterns = [
    url(r'list',views.blog_view_list),
    url(r'views-view',views.views_view)
]