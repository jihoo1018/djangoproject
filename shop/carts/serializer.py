from rest_framework import serializers
from shop.carts.models import Cart


class CartSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cart
        fields = '__all__'

    def create(self, validated_data):
        return Cart.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Cart.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass
