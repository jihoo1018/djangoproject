from rest_framework import serializers
from multiplex.movie_users.models import MovieUser


class MovieUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = MovieUser
        fields = '__all__'

    def create(self, validated_data):
        return MovieUser.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        MovieUser.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass
