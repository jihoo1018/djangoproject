from rest_framework import serializers
from multiplex.cinemas.models import Cinema


class CinemaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cinema
        fields = '__all__'

    def create(self, validated_data):
        return Cinema.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Cinema.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass
