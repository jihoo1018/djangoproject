from rest_framework import serializers
from shop.deliveries.models import Delivery


class DeliverySerializer(serializers.ModelSerializer):
    class Meta:
        model = Delivery
        fields = '__all__'

    def create(self, validated_data):
        return Delivery.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Delivery.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass

