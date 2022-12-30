from rest_framework import serializers
from shop.shop_users.models import ShopUser


class ShopUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShopUser
        fields = '__all__'

    def create(self, validated_data):
        return ShopUser.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        ShopUser.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass

