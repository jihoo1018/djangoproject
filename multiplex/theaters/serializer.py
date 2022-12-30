from rest_framework import serializers
from multiplex.theaters.models import Theater


class TheaterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Theater
        fields = '__all__'

    def create(self, validated_data):
        return Theater.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Theater.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass
