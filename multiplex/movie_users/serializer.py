from rest_framework import serializers
from multiplex.movie_users.models import MovieUser


class MovieUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = MovieUser
        fields = '__all__'
