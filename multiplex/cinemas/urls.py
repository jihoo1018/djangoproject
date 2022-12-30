from django.urls import re_path as url

from multiplex.cinemas import views

urlpatterns = [
    url(r'list',views.cinema_list),
    url(r'cinema-view',views.cinema_view)
]