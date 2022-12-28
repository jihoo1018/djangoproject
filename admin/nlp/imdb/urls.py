from django.urls import re_path as url
from admin.nlp.imdb import views

urlpatterns = [
    url(r'moviereview',views.movie_review)
]