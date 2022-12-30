from rest_framework import serializers
from shop.products.models import Product


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'

    def create(self, validated_data):
        return Product.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Product.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass
