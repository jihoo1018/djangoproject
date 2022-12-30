from django.urls import re_path as url

from shop.deliveries import views

urlpatterns = [
    url(r'list',views.delivery_list),
    url(r'delivery-view',views.delivery_view)
]