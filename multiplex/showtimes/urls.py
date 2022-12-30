from django.urls import re_path as url

from multiplex.showtimes import views

urlpatterns = [
    url(r'list',views.showtimes_list),
    url(r'showtimes-view',views.showtimes_view)
]