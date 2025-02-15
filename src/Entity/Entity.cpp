#include "Entity.h"

Entity::Entity(const uint &t_id):m_id(t_id) {
}

Void Entity::outdate() {
    m_updated = false;
}

Void Entity::update(const Point &t_position, const Float &t_angle) {
    m_position = t_position;
    m_angle = t_angle;
    m_updated = true;
}

String Entity::to_string() {
    return "Robot ID " + std::to_string(m_id) 
      + " from team " + std::to_string(m_team)
      + " at (" + std::to_string(m_position.x) + ", " 
      + std::to_string(m_position.y) + ") with angle " 
      + std::to_string(m_angle) + " radians"; 
}

Point Entity::position() {
    return m_position;
}

Bool Entity::updated()
{
  return m_updated;
}

Float Entity::angle()
{
  return m_angle;
}

void Entity::id(const uint& t_id) {
    m_id = t_id;
}

const uint &Entity::id()
{
  return m_id;
}

void Entity::team(const uint& t_team) {
    m_team = t_team;
}

const uint &Entity::team()
{
  return m_team;
}

Bool operator<(Entity& a, Entity& b) {
    return a.id() < b.id();
}
