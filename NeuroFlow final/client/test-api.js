// Quick test script to check API connection
async function testAPI() {
    try {
        console.log('Testing API connection...');
        const response = await fetch('http://localhost:8000/api/emotions/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: 'I am happy' })
        });
        
        console.log('Response status:', response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log('Success! Response:', data);
        } else {
            console.log('Error response:', await response.text());
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
}

testAPI();