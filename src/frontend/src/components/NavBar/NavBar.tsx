import React, { useState } from 'react';

function SearchBar({ onSearch }) {
    const [searchTerm, setSearchTerm] = useState('');
    const [results, setResults] = useState([]);

    const handleInputChange = (event) => {
        const term = event.target.value
        setSearchTerm(searchTerm);
        if (term) {
            onSearch(term).then((data) => setResults(data));
        } else {
            setResults([]);
        }
    }

    const handleSubmit = (event) => {
        event.preventDefault()
    }

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="text"
                placeholder='Search...'
                value={searchTerm}
                onChange={handleInputChange}
            />
            <button type="submit">Search</button>
            {results.length > 0 && (
                <ul>
                    {results.map((company: any, index) => ( // get rid of any
                        <li key={index}>{company.name}</li>
                    ))}
                </ul>
            )}
        </form>
    )

}


export default SearchBar