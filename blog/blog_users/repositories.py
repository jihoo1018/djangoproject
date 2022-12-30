from django.http import JsonResponse
from blog.blog_users.models import BlogUser
from blog.blog_users.serializer import BlogUserSerializer
from rest_framework.response import Response

class UserRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = BlogUserSerializer(BlogUser.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self,param):
        return BlogUser.objects.all().filter(blog_userid=param).values()[0]

    def login(self, kwargs):
        loginuser = BlogUser.objects.get(email=kwargs['email'])
        if loginuser.password == kwargs["password"]:
            dbuser = BlogUser.objects.all().filter(blog_userid=loginuser.blog_userid).values()[0]
            serialize = BlogUserSerializer(dbuser, many=False)
            return JsonResponse(data=serialize.data, safe=False)
        # dictionary이외를 받을 경우, 두번째 argument를 safe=False로 설정해야한다.
        else:
            return JsonResponse("비번이 틀립니다")

    def find_by_email(self, param):
        return BlogUser.objects.all().filter(email=param).values()[0]

    def find_by_name(self, param):
        return BlogUser.objects.all().filter(nickname=param).values()[0]



