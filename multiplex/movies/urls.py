from django.urls import re_path as url
from multiplex.movies import views

urlpatterns = [
    url(r'fake-faces',views.fake_face),
    url(r'list',views.movies_list),
    url(r'movies-view',views.movies_view)
]