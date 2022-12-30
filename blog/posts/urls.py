from django.urls import re_path as url

from blog.posts import views

urlpatterns = [
    url(r'list',views.post_list),
    url(r'post-view',views.post_view)
]