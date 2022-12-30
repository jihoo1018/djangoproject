from rest_framework.response import Response

from shop.shop_users.models import ShopUser
from shop.shop_users.serializer import ShopUserSerializer


class ShopUserRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = ShopUserSerializer(ShopUser.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return