from blog.comments import views
from django.urls import re_path as url

urlpatterns = [
    url(r'list',views.comment_list),
    url(r'comment-view',views.comment_view)
]