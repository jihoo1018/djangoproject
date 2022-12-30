from django.urls import re_path as url

from shop.shop_users import views

urlpatterns = [
    url(r'list',views.user_list),
    url(r'shop-user-view',views.shop_user_view)
]