from django.urls import re_path as url

from multiplex.theaters import views

urlpatterns = [
    url(r'list',views.theater_list),
    url(r'theater-view',views.theaters_view)
]