from django.urls import re_path as url

from shop.carts import views

urlpatterns = [
    url(r'list',views.cart_list),
    url(r'cart-view',views.cart_view)
]