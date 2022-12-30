from django.urls import re_path as url

from blog.tags import views

urlpatterns = [
    url(r'list',views.tags_list),
    url(r'tag-view',views.tag_view)
]