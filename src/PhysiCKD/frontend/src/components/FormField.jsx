import React from 'react';

const FormField = ({ label, id, type = 'text', options = [], value, onChange, error, required, ...props }) => {
    return (
        <div className="form-field">
            <label htmlFor={id}>
                {label}
                {required && <span className="required-indicator"> *</span>}
            </label>
            {type === 'select' ? (
                <select 
                    id={id} 
                    name={id} 
                    value={value} 
                    onChange={onChange} 
                    required={required}
                    className={error ? 'error' : ''}
                    {...props}
                >
                    <option value="">Select...</option>
                    {options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                            {opt.label}
                        </option>
                    ))}
                </select>
            ) : (
                <input
                    type={type}
                    id={id}
                    name={id}
                    value={value}
                    onChange={onChange}
                    required={required}
                    className={error ? 'error' : ''}
                    {...props}
                />
            )}
            {error && <span className="field-error">{error}</span>}
        </div>
    );
};

export default FormField;
