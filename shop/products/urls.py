from django.urls import re_path as url

from shop.products import views

urlpatterns = [
    url(r'list',views.product_list),
    url(r'product-view',views.product_view)
]