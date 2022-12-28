from rest_framework import serializers
from multiplex.showtimes.models import Showtime


class ShowtimeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Showtime
        fields = '__all__'
