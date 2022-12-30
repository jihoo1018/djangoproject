from rest_framework import serializers
from multiplex.showtimes.models import Showtime


class ShowtimeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Showtime
        fields = '__all__'

    def create(self,validated_data):
        return Showtime.objects.create(**validated_data)

    def update(self,instace,valicated_data):
        Showtime.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self,instance,valicated_data):
        pass

