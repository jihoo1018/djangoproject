from django.urls import re_path as url

from multiplex.theater_tickets import views

urlpatterns = [
    url(r'list',views.theater_tickets_list),
    url(r'theater-ticket-view',views.theater_ticket_view)
]