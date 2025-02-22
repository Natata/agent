from dataclasses import dataclass
from uuid import uuid4, UUID
from typing import Dict, List, Any


@dataclass
class Customer:
    id: UUID
    name: str


class UserNotFoundError(Exception):
    """Raised when a user with the given ID is not found."""
    pass


class CustomerServiceClient:
    def __init__(self) -> None:
        # In-memory storage for customers.
        # Keys are customer UUIDs, values are Customer objects.
        self._customers: Dict[UUID, Customer] = {}

    def create_customer(self, name: str) -> Customer:
        """
        Create a new customer with the given name.
        The customer's ID is auto-generated.
        """
        new_id = uuid4()
        customer = Customer(id=new_id, name=name)
        self._customers[new_id] = customer
        return customer

    def get_customer(self, customer_id: UUID) -> Customer:
        """
        Retrieve a customer by ID.
        Raises UserNotFoundError if the customer does not exist.
        """
        if customer_id in self._customers:
            return self._customers[customer_id]
        else:
            raise UserNotFoundError(f"customer with id {customer_id} not found.")

    def list_customers(self, offset: int, limit: int) -> Dict[str, Any]:
        """
        List customers with pagination.
        
        Parameters:
            offset (int): The starting index.
            limit (int): The number of customers to return; allowed values are 1, 2, or 3.
        
        Returns:
            A dictionary with:
                - "customer": a list of Customer objects,
                - "offset": the next offset to use (or 0 if no more customers),
                - "limit": the limit to use for the next query (or 0 if no more customers).
        
        If the offset exceeds the number of stored customers, returns an empty list and both offset and limit are 0.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit not in (1, 2, 3):
            raise ValueError("limit must be 1, 2, or 3")

        all_customers: List[Customer] = list(self._customers.values())
        total_customers = len(all_customers)

        if offset >= total_customers:
            return {"customers": [], "offset": 0, "limit": 0}

        # Slice the list to get the desired page.
        sub_customers = all_customers[offset: offset + limit]
        new_offset = offset + len(sub_customers)

        # If we've reached or exceeded the total number of customers, reset offset and limit.
        if new_offset >= total_customers:
            next_offset = 0
            next_limit = 0
        else:
            next_offset = new_offset
            next_limit = limit

        return {"customers": sub_customers, "offset": next_offset, "limit": next_limit}

    def update_customer(self, customer: Customer) -> Customer:
        """
        Update the name of an existing customer.
        
        Parameters:
            customer (Customer): A Customer object with the updated name. The customer's ID is used to locate the record.
        
        Returns:
            The updated Customer object.
        
        Raises:
            UserNotFoundError if the user does not exist.
        """
        if customer.id not in self._customers:
            raise UserNotFoundError(f"User with id {customer.id} not found.")

        # Only the name is updated.
        self._customers[customer.id].name = customer.name
        return self._customers[customer.id]

    def delete_customer(self, customer_id: UUID) -> None:
        """
        Delete a customer by ID.
        
        Raises:
            UserNotFoundError if the customer does not exist.
        """
        if customer_id not in self._customers:
            raise UserNotFoundError(f"Customer with id {customer_id} not found.")
        del self._customers[customer_id]


# Example usage (for testing purposes):
if __name__ == "__main__":
    client = CustomerServiceClient()
    
    # Create some customers
    customer1 = client.create_customer("Alice")
    customer2 = client.create_customer("Bob")
    customer3 = client.create_customer("Charlie")
    
    # Read a customer
    try:
        print("Get customer1:", client.get_customer(customer1.id))
    except UserNotFoundError as e:
        print(e)
    
    # List customers with pagination
    page1 = client.list_customers(offset=0, limit=2)
    print("Page 1:", page1)
    
    # If there is a next page, list next customers.
    if page1["offset"] != 0:
        page2 = client.list_customers(offset=page1["offset"], limit=page1["limit"])
        print("Page 2:", page2)
    
    # Update a customer
    customer1.name = "Alicia"
    updated_customer1 = client.update_customer(customer1)
    print("Updated customer1:", updated_customer1)
    
    # Delete a customer
    client.delete_customer(customer2.id)
    try:
        client.get_customer(customer2.id)
    except UserNotFoundError as e:
        print("After deletion:", e)
