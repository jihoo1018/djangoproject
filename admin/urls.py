"""admin URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from admin.views import hello

urlpatterns = [
    path('', hello),
    path("blog/auth/",include('blog.blog_users.urls')),
    path("mplex/movies/",include('multiplex.movies.urls')),
    path("blog_/",include('blog.urls')),
    path("shop/",include('shop.urls')),
    path("admin/dlearn/",include('admin.dlearn.urls')),
    path("admin/dlearn/webcrawler/", include('admin.dlearn.webcrawler.url')),
    path("admin/nlp/", include('admin.nlp.urls')),
    path("admin/nlp/imdb/", include('admin.nlp.imdb.urls'))
]
