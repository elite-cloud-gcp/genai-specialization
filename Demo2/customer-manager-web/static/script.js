// Get references to necessary HTML elements
const customerContainer = document.getElementById('customerContainer');
const searchInput = document.getElementById('searchInput');

// API endpoint for fetching customer data
const apiEndpoint = 'https://customer-api-xxxxxxx-uc.a.run.app/api/customers'; // Replace with your own API endpoint. Don't forget the /api/customers route !

// Object to store all customer data
let customers = {}; 


// --- Fetch and Display Customers ---

// Fetch customer data from the API
async function fetchCustomers() {
    try {
        const response = await fetch(apiEndpoint);
        customers = await response.json(); // Store fetched data
        displayCustomers(customers); // Display all customers initially
    } catch (error) {
        console.error('Error fetching customers:', error);
    }
}

// Display customer cards in the container
function displayCustomers(customersToDisplay) {
    customerContainer.innerHTML = ''; // Clear existing cards

    for (const customerId in customersToDisplay) {
        const customer = customersToDisplay[customerId];
        createCustomerCard(customer); // Create a card for each customer
    }
}


// --- Create and Filter Customer Cards ---

// Create a customer card element with their details
function createCustomerCard(customer) {
    const card = document.createElement('div');
    card.classList.add('customerCard');
    card.innerHTML = `
        <h3>${customer.firstName} ${customer.lastName}</h3>
        <p>ID: ${customer.customer_id}</p>
        <p>Email: ${customer.email}</p>
        <p>Phone: ${customer.phone}</p>
        <p>Address: ${customer.address.street}, ${customer.address.city}, ${customer.address.state}, ${customer.address.postalCode}, ${customer.address.country}</p>
        <p>Teletransmission: ${customer.teletransmission ? 'Yes' : 'No'}</p>  
        <p>Surcomplementaire: ${customer.surcomplementaire ? 'Yes' : 'No'}</p>
        <p>Member Card Issued: ${customer.MemberCardIssuanceDate ? new Date(customer.MemberCardIssuanceDate).toLocaleDateString() : "N/A"}</p>
        <button onclick="viewCustomerDetails('${customer.customer_id}')">View Details</button> 
    `;
    customerContainer.appendChild(card);
}

// Placeholder for a function to display detailed customer info
function viewCustomerDetails(customerId) {
    console.log("View details for customer ID:", customerId); 
}

// Filter customers based on search term
function filterCustomers(searchTerm) {
    const filteredCustomers = {};
    for (const customerId in customers) {
        const customer = customers[customerId];
        const name = `${customer.firstName} ${customer.lastName}`.toLowerCase();
        if (name.includes(searchTerm) || customerId.includes(searchTerm)) {
            filteredCustomers[customerId] = customer;
        }
    }
    displayCustomers(filteredCustomers); // Display filtered results
}

// --- Search Input Event Listener ---

// Update displayed customers when search input changes
searchInput.addEventListener('input', () => {
    const searchTerm = searchInput.value.toLowerCase();
    filterCustomers(searchTerm);
});


// --- Initial Data Fetch ---

// Fetch customer data when the page loads
fetchCustomers(); 
