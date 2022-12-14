from django.urls import re_path as url

from shop import iris_view

urlpatterns = [
    url(r'iris',iris_view.iris)
]