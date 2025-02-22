from dataclasses import dataclass
from uuid import uuid4, UUID
from typing import Dict, List, Any


@dataclass
class User:
    id: UUID
    name: str


class UserNotFoundError(Exception):
    """Raised when a user with the given ID is not found."""
    pass


class ServiceAClient:
    def __init__(self) -> None:
        # In-memory storage for users.
        # Keys are user UUIDs, values are User objects.
        self._users: Dict[UUID, User] = {}

    def create_user(self, name: str) -> User:
        """
        Create a new user with the given name.
        The user's ID is auto-generated.
        """
        new_id = uuid4()
        user = User(id=new_id, name=name)
        self._users[new_id] = user
        return user

    def get_user(self, user_id: UUID) -> User:
        """
        Retrieve a user by ID.
        Raises UserNotFoundError if the user does not exist.
        """
        if user_id in self._users:
            return self._users[user_id]
        else:
            raise UserNotFoundError(f"User with id {user_id} not found.")

    def list_users(self, offset: int, limit: int) -> Dict[str, Any]:
        """
        List users with pagination.
        
        Parameters:
            offset (int): The starting index.
            limit (int): The number of users to return; allowed values are 1, 2, or 3.
        
        Returns:
            A dictionary with:
                - "users": a list of User objects,
                - "offset": the next offset to use (or 0 if no more users),
                - "limit": the limit to use for the next query (or 0 if no more users).
        
        If the offset exceeds the number of stored users, returns an empty list and both offset and limit are 0.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit not in (1, 2, 3):
            raise ValueError("limit must be 1, 2, or 3")

        all_users: List[User] = list(self._users.values())
        total_users = len(all_users)

        if offset >= total_users:
            return {"users": [], "offset": 0, "limit": 0}

        # Slice the list to get the desired page.
        sub_users = all_users[offset: offset + limit]
        new_offset = offset + len(sub_users)

        # If we've reached or exceeded the total number of users, reset offset and limit.
        if new_offset >= total_users:
            next_offset = 0
            next_limit = 0
        else:
            next_offset = new_offset
            next_limit = limit

        return {"users": sub_users, "offset": next_offset, "limit": next_limit}

    def update_user(self, user: User) -> User:
        """
        Update the name of an existing user.
        
        Parameters:
            user (User): A User object with the updated name. The user's ID is used to locate the record.
        
        Returns:
            The updated User object.
        
        Raises:
            UserNotFoundError if the user does not exist.
        """
        if user.id not in self._users:
            raise UserNotFoundError(f"User with id {user.id} not found.")

        # Only the name is updated.
        self._users[user.id].name = user.name
        return self._users[user.id]

    def delete_user(self, user_id: UUID) -> None:
        """
        Delete a user by ID.
        
        Raises:
            UserNotFoundError if the user does not exist.
        """
        if user_id not in self._users:
            raise UserNotFoundError(f"User with id {user_id} not found.")
        del self._users[user_id]


# Example usage (for testing purposes):
if __name__ == "__main__":
    client = ServiceAClient()
    
    # Create some users
    user1 = client.create_user("Alice")
    user2 = client.create_user("Bob")
    user3 = client.create_user("Charlie")
    
    # Read a user
    try:
        print("Get user1:", client.get_user(user1.id))
    except UserNotFoundError as e:
        print(e)
    
    # List users with pagination
    page1 = client.list_users(offset=0, limit=2)
    print("Page 1:", page1)
    
    # If there is a next page, list next users.
    if page1["offset"] != 0:
        page2 = client.list_users(offset=page1["offset"], limit=page1["limit"])
        print("Page 2:", page2)
    
    # Update a user
    user1.name = "Alicia"
    updated_user1 = client.update_user(user1)
    print("Updated user1:", updated_user1)
    
    # Delete a user
    client.delete_user(user2.id)
    try:
        client.get_user(user2.id)
    except UserNotFoundError as e:
        print("After deletion:", e)
