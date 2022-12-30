from django.urls import re_path as url

from multiplex.movie_users import views

urlpatterns = [
    url(r'list',views.user_list),
    url(r'movie-user-view',views.movie_user_view)
]