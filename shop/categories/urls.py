from django.urls import re_path as url

from shop.categories import views

urlpatterns = [
    url(r'list',views.category_list),
    url(r'category-view',views.catgories_view)
]