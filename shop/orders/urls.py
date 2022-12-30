from django.urls import re_path as url

from shop.orders import views

urlpatterns = [
    url(r'list',views.order_list),
    url(r'order-view',views.order_view)
]